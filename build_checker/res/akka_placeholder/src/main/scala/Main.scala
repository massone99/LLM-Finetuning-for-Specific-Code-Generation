import akka.actor.{Actor, ActorSystem, Props}

case object Toggle
case object Reset
case class Msg(content: String)

class ToggleActor extends Actor {
  // Start in active state
  def receive: Receive = active

  def active: Receive = {
    case Toggle =>
      println("ToggleActor: Switching from active to inactive.")
      context.become(inactive)
    case Reset =>
      println("ToggleActor: Already active, staying active.")
      // Remain in the same active state
    case Msg(content) =>
      println(s"ToggleActor (active) got -> $content")
    case _ =>
      println("ToggleActor (active) received unknown message.")
  }

  def inactive: Receive = {
    case Toggle =>
      println("ToggleActor: Switching from inactive to active.")
      context.become(active)
    case Reset =>
      println("ToggleActor: Reset invoked, going to active.")
      context.become(active)
    case Msg(content) =>
      println(s"ToggleActor (inactive) got -> $content")
    case _ =>
      println("ToggleActor (inactive) received unknown message.")
  }
}

object ToggleApp extends App {
  val system = ActorSystem("ToggleSystem")
  val toggler = system.actorOf(Props[ToggleActor](), "toggleActor")

  toggler ! Msg("Hello in active mode.")
  toggler ! Toggle
  toggler ! Msg("Now I'm in inactive mode.")

  // Use Reset to go back to active no matter what
  toggler ! Reset
  toggler ! Msg("Back in active mode.")

  Thread.sleep(500)
  system.terminate()
}