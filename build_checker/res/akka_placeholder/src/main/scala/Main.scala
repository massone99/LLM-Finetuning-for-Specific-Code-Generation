import akka.actor.{Actor, ActorSystem, Props}
import akka.stream.ActorMaterializer
import akka.stream.scaladsl.{Source, Sink}

class StreamProcessingActor extends Actor {
  implicit val materializer: ActorMaterializer = ActorMaterializer()(context.system)
  def receive: Receive = {
    case "start" =>
      Source(1 to 10).runWith(Sink.foreach(n => println(s"StreamProcessingActor received: $n")))
  }
}

object StreamProcessingApp extends App {
  val system = ActorSystem("StreamProcessingSystem")
  val streamActor = system.actorOf(Props[StreamProcessingActor](), "streamActor")
  streamActor ! "start"
  Thread.sleep(1000)
  system.terminate()
}